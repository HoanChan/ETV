import os
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.metric.post_processing import (
    text_to_list,
    htmlPostProcess,
    deal_isolate_span,
    deal_duplicate_bb,
    deal_bb,
    merge_span_token,
    deal_eb_token,
    insert_text_to_token
)

"""
All bugs have been fixed in the implementation:

1. insert_text_to_token() now creates valid HTML by skipping tokens without content
2. deal_duplicate_bb() now properly balances <b></b> tags with span attributes  
3. merge_span_token() now handles empty input gracefully without IndexError

Tests now verify correct behavior instead of accepting bugs.
"""


# Test data for text_to_list function
@pytest.mark.parametrize("master_token, expected", [
    # Test basic token splitting with comma separator
    ("<td>cell1</td>,<td>cell2</td>", ["<td>cell1</td>", "<td>cell2</td>"]),
    # Test when master token ends with <td></td>
    ("<td>cell1</td>,<td></td>", ["<td>cell1</td>", "<td></td>"]),
    # Test when master token already ends with </tbody>
    ("<td>cell1</td>,</tbody>", ["<td>cell1</td>", "</tbody>"]),
    # Test when master token already has </tr> and </tbody>
    ("<td>cell1</td>,</tr>,</tbody>", ["<td>cell1</td>", "</tr>", "</tbody>"]),
    # Test with empty token
    ("", [""]),
    # Test with single token
    ("<td>single</td>", ["<td>single</td>"]),
])
def test_text_to_list(master_token, expected):
    """Test text_to_list function with various inputs."""
    result = text_to_list(master_token)
    assert result == expected


# Test data for htmlPostProcess function
@pytest.mark.parametrize("text, expected", [
    # Test basic HTML wrapping
    ("<tr><td>content</td></tr>", "<html><body><table><tr><td>content</td></tr></table></body></html>"),
    # Test with empty text
    ("", "<html><body><table></table></body></html>"),
    # Test with complex table structure
    ("<thead><tr><td>Header</td></tr></thead><tbody><tr><td>Data</td></tr></tbody>", 
     "<html><body><table><thead><tr><td>Header</td></tr></thead><tbody><tr><td>Data</td></tr></tbody></table></body></html>"),
])
def test_htmlPostProcess(text, expected):
    """Test htmlPostProcess function with various inputs."""
    result = htmlPostProcess(text)
    assert result == expected


# Additional edge cases for htmlPostProcess
@pytest.mark.parametrize("text, expected", [
    # Test with special characters and entities
    ("&lt;script&gt;alert('test')&lt;/script&gt;", 
     "<html><body><table>&lt;script&gt;alert('test')&lt;/script&gt;</table></body></html>"),
    # Test with unicode characters
    ("café naïve résumé", "<html><body><table>café naïve résumé</table></body></html>"),
    # Test with newlines and whitespace
    ("\n\t<tr>\n\t\t<td>content</td>\n\t</tr>\n", 
     "<html><body><table>\n\t<tr>\n\t\t<td>content</td>\n\t</tr>\n</table></body></html>"),
])
def test_htmlPostProcess_edge_cases(text, expected):
    """Test htmlPostProcess function with edge cases."""
    result = htmlPostProcess(text)
    assert result == expected


# Test data for deal_isolate_span function
@pytest.mark.parametrize("thead_part, expected", [
    # Test fixing isolate rowspan and colspan
    ('<td></td> rowspan="2" colspan="3"></b></td>', '<td rowspan="2" colspan="3"></td>'),
    # Test fixing isolate colspan and rowspan
    ('<td></td> colspan="3" rowspan="2"></b></td>', '<td colspan="3" rowspan="2"></td>'),
    # Test fixing isolate rowspan only
    ('<td></td> rowspan="2"></b></td>', '<td rowspan="2"></td>'),
    # Test fixing isolate colspan only
    ('<td></td> colspan="3"></b></td>', '<td colspan="3"></td>'),
    # Test with no isolate spans
    ('<td>normal content</td>', '<td>normal content</td>'),
    # Test with multiple isolate spans
    ('<td></td> rowspan="2"></b></td><td></td> colspan="3"></b></td>', 
     '<td rowspan="2"></td><td colspan="3"></td>'),
])
def test_deal_isolate_span(thead_part, expected):
    """Test deal_isolate_span function with various inputs."""
    result = deal_isolate_span(thead_part)
    assert result == expected


# Additional edge cases for deal_isolate_span
@pytest.mark.parametrize("thead_part, expected", [
    # Test with large span numbers
    ('<td></td> rowspan="99" colspan="88"></b></td>', '<td rowspan="99" colspan="88"></td>'),
    # Test with no isolate spans (should return unchanged)
    ('<td>normal content</td><td>more content</td>', '<td>normal content</td><td>more content</td>'),
    # Test with empty input
    ('', ''),
])
def test_deal_isolate_span_edge_cases(thead_part, expected):
    """Test deal_isolate_span function with edge cases."""
    result = deal_isolate_span(thead_part)
    assert result == expected


# Test data for deal_duplicate_bb function
@pytest.mark.parametrize("thead_part, expected", [
    # Test removing multiple bold tags
    ('<td><b>text<b>more</b>text</b></td>', '<td><b>textmoretext</b></td>'),
    # Test with single bold tags (no change)
    ('<td><b>text</b></td>', '<td><b>text</b></td>'),
    # Test with no bold tags
    ('<td>text</td>', '<td>text</td>'),
    # FIXED: Test with span attributes - now properly balances <b></b> tags
    ('<td rowspan="2"><b>text<b>more</b></b></td>', '<td rowspan="2"><b>textmore</b></td>'),
])
def test_deal_duplicate_bb(thead_part, expected):
    """Test deal_duplicate_bb function with various inputs."""
    result = deal_duplicate_bb(thead_part)
    assert result == expected


# Test data for deal_bb function
@pytest.mark.parametrize("result_token, tag, expected", [
    # Test thead processing without spans
    ('<thead><tr><td>Header</td></tr></thead>', 'thead', '<thead><tr><td><b>Header</b></td></tr></thead>'),
    # Test thead processing with spans
    ('<thead><tr><td rowspan="2">Header</td></tr></thead>', 'thead', '<thead><tr><td rowspan="2"><b>Header</b></td></tr></thead>'),
    # Test tbody processing (only isolate span handling)
    ('<tbody><tr><td></td> rowspan="2"></b></td></tr></tbody>', 'tbody', '<tbody><tr><td rowspan="2"></td></tr></tbody>'),
    # Test when no matching tag found
    ('<tr><td>content</td></tr>', 'thead', '<tr><td>content</td></tr>'),
    # Test with empty cells
    ('<thead><tr><td></td><td>Header</td></tr></thead>', 'thead', '<thead><tr><td></td><td><b>Header</b></td></tr></thead>'),
])
def test_deal_bb(result_token, tag, expected):
    """Test deal_bb function with various inputs."""
    result = deal_bb(result_token, tag)
    assert result == expected


# Additional edge cases for deal_bb  
@pytest.mark.parametrize("result_token, tag, expected", [
    # Test with empty thead
    ('<thead></thead>', 'thead', '<thead></thead>'),
    # Test with complex nested bold tags
    ('<thead><tr><td><b>text1</b></td><td>text2<b>bold</b>normal</td></tr></thead>', 'thead', 
     '<thead><tr><td><b>text1</b></td><td><b>text2boldnormal</b></td></tr></thead>'),
    # Test tbody with mixed content and isolate spans  
    ('<tbody><tr><td>content</td><td></td> rowspan="2"></b></td></tr></tbody>', 'tbody',
     '<tbody><tr><td>content</td><td rowspan="2"></td></tr></tbody>'),
    # Test with no td tags
    ('<thead><tr></tr></thead>', 'thead', '<thead><tr></tr></thead>'),
])
def test_deal_bb_edge_cases(result_token, tag, expected):
    """Test deal_bb function with edge cases."""
    result = deal_bb(result_token, tag)
    assert result == expected


# Test data for merge_span_token function
@pytest.mark.parametrize("master_token_list, expected", [
    # Test merging rowspan and colspan tokens
    (['<td', ' rowspan="2"', ' colspan="3"', '>', '</td>'], ['<td rowspan="2" colspan="3"></td>', '</tbody>']),
    # Test merging single span token
    (['<td', ' rowspan="2"', '>', '</td>'], ['<td rowspan="2"></td>', '</tbody>']),
    # Test with no span tokens
    (['<td>', 'content', '</td>'], ['<td>', 'content', '</td>', '</tbody>']),
    # Test with mixed span and regular tokens - span merges but content gets separated
    (['<td>', 'cell1', '</td>', '<td', ' colspan="2"', '>', 'cell2', '</td>'], 
     ['<td>', 'cell1', '</td>', '<td colspan="2">cell2', '</td>', '</tbody>']),
    # Test when list already ends with tbody
    (['<td>content</td>', '</tbody>'], ['<td>content</td>', '</tbody>']),
    # FIXED: Empty list test - now handles gracefully without IndexError
    ([], ['</tbody>']),
])
def test_merge_span_token(master_token_list, expected):
    """Test merge_span_token function with various inputs."""
    result = merge_span_token(master_token_list)
    assert result == expected


# Additional edge cases for merge_span_token
@pytest.mark.parametrize("master_token_list, expected", [
    # Test incomplete span tokens (edge case that shouldn't normally happen)
    (['<td', ' rowspan="2"'], ['<td', ' rowspan="2"', '</tbody>']),
    # Test with only non-td tokens
    (['<tr>', '</tr>', '<tbody>'], ['<tr>', '</tr>', '<tbody>', '</tbody>']),
    # Test span tokens followed by other attributes (merges fixed 5-token pattern)
    (['<td', ' rowspan="2"', ' colspan="3"', ' id="test"', '>', 'content', '</td>'], 
     ['<td rowspan="2" colspan="3" id="test">', 'content', '</td>', '</tbody>']),
    # Test multiple span attributes together
    (['<td', ' colspan="2"', ' rowspan="3"', '>', '</td>'], 
     ['<td colspan="2" rowspan="3"></td>', '</tbody>']),
])
def test_merge_span_token_edge_cases(master_token_list, expected):
    """Test merge_span_token function with edge cases.""" 
    result = merge_span_token(master_token_list)
    assert result == expected


# Test data for deal_eb_token function
@pytest.mark.parametrize("master_token, expected", [
    # Test basic empty bbox token replacements
    ('<eb></eb><eb1></eb1><eb2></eb2>', '<td></td><td> </td><td><b> </b></td>'),
    # Test all empty bbox token types
    ('<eb></eb><eb1></eb1><eb2></eb2><eb3></eb3><eb4></eb4><eb5></eb5><eb6></eb6><eb7></eb7><eb8></eb8><eb9></eb9><eb10></eb10>', 
     '<td></td><td> </td><td><b> </b></td><td>\u2028\u2028</td><td><sup> </sup></td><td><b></b></td><td><i> </i></td><td><b><i></i></b></td><td><b><i> </i></b></td><td><i></i></td><td><b> \u2028 \u2028 </b></td>'),
    # Test with no empty bbox tokens
    ('<td>regular content</td>', '<td>regular content</td>'),
    # Test with mixed content and eb tokens
    ('<td>content</td><eb></eb><td>more content</td>', '<td>content</td><td></td><td>more content</td>'),
])
def test_deal_eb_token(master_token, expected):
    """Test deal_eb_token function with various inputs."""
    result = deal_eb_token(master_token)
    assert result == expected


# Additional edge cases for deal_eb_token
@pytest.mark.parametrize("master_token, expected", [
    # Test mixed eb tokens with regular HTML
    ('<tr><eb></eb><td>content</td><eb1></eb1></tr>', '<tr><td></td><td>content</td><td> </td></tr>'),
    # Test consecutive same eb tokens
    ('<eb></eb><eb></eb><eb1></eb1>', '<td></td><td></td><td> </td>'),
    # Test eb tokens at start and end
    ('<eb></eb><td>middle</td><eb2></eb2>', '<td></td><td>middle</td><td><b> </b></td>'),
    # Test all eb tokens in sequence
    ('<eb></eb><eb1></eb1><eb2></eb2><eb3></eb3><eb4></eb4><eb5></eb5><eb6></eb6><eb7></eb7><eb8></eb8><eb9></eb9><eb10></eb10>',
     '<td></td><td> </td><td><b> </b></td><td>\u2028\u2028</td><td><sup> </sup></td><td><b></b></td><td><i> </i></td><td><b><i></i></b></td><td><b><i> </i></b></td><td><i></i></td><td><b> \u2028 \u2028 </b></td>'),
])
def test_deal_eb_token_edge_cases(master_token, expected):
    """Test deal_eb_token function with edge cases."""
    result = deal_eb_token(master_token)
    assert result == expected


# Test data for insert_text_to_token function
@pytest.mark.parametrize("master_token_list, cell_content_list, expected", [
    # Test basic text insertion into tokens
    (['<td>', '</td>', '<td>', '</td>'], ['cell1', 'cell2'], '<td>cell1</td><td>cell2</td></tbody>'),
    # Test text insertion with span tokens
    (['<td', ' rowspan="2"', '>', '</td>'], ['spanned cell'], '<td rowspan="2">spanned cell</td></tbody>'),
    # FIXED: Test when there are fewer cell contents than tokens - now skips extra tokens for valid HTML
    (['<td>', '</td>', '<td>', '</td>', '<td>', '</td>'], ['cell1', 'cell2'], '<td>cell1</td><td>cell2</td></tbody>'),
    # Test when there are more cell contents than tokens - only processes available tokens  
    (['<td>', '</td>'], ['cell1', 'cell2', 'cell3'], '<td>cell1</td></tbody>'),
    # Test text insertion with empty bbox tokens - eb tokens get processed, content inserted
    (['<td>', '</td>', '<eb></eb>'], ['content'], '<td>content</td><td></td></tbody>'),
    # FIXED: Test with empty cell content list - now produces valid HTML (empty result)
    (['<td>', '</td>'], [], '</tbody>'),
    # Test with non-td tokens - content inserted into td tokens
    (['<tr>', '<td>', '</td>', '</tr>'], ['content'], '<tr><td>content</td></tr></tbody>'),
])
def test_insert_text_to_token(master_token_list, cell_content_list, expected):
    """Test insert_text_to_token function with various inputs."""
    result = insert_text_to_token(master_token_list, cell_content_list)
    assert result == expected


# Additional edge cases for insert_text_to_token  
@pytest.mark.parametrize("master_token_list, cell_content_list, expected", [
    # Test mixed self-contained and separate tokens
    (['<td></td>', '<td', ' rowspan="2"', '>', '</td>'], ['cell1', 'cell2'], 
     '<td>cell1</td><td rowspan="2">cell2</td></tbody>'),
    # Test with only non-td tokens (should just return structure)
    (['<tr>', '</tr>', '<tbody>'], ['content'], '<tr></tr><tbody></tbody>'),
    # Test eb tokens at beginning
    (['<eb1></eb1>', '<td>', '</td>'], ['content'], '<td> </td><td>content</td></tbody>'),
    # Test complex mixed structure
    (['<tr>', '<td>', '</td>', '<eb></eb>', '<td', ' colspan="2"', '>', '</td>', '</tr>'], 
     ['cell1', 'cell2'], '<tr><td>cell1</td><td></td><td colspan="2">cell2</td></tr></tbody>'),
])
def test_insert_text_to_token_edge_cases(master_token_list, cell_content_list, expected):
    """Test insert_text_to_token function with edge cases."""
    result = insert_text_to_token(master_token_list, cell_content_list)
    assert result == expected


# Additional edge cases for text_to_list
@pytest.mark.parametrize("master_token, expected", [
    # Test with only commas
    (",,", ["", "", ""]),
    # Test complex content with single quotes
    ("<td>cell1</td>,<td colspan='2'>cell2</td>,<td></td>", 
     ["<td>cell1</td>", "<td colspan='2'>cell2</td>", "<td></td>"]),
    # Test with special characters
    ("<td>café</td>,<td>naïve</td>", ["<td>café</td>", "<td>naïve</td>"]),
    # Test with HTML entities
    ("<td>&lt;test&gt;</td>,<td>&amp;data</td>", ["<td>&lt;test&gt;</td>", "<td>&amp;data</td>"]),
])
def test_text_to_list_edge_cases(master_token, expected):
    """Test text_to_list function with edge cases."""
    result = text_to_list(master_token)
    assert result == expected


# Integration tests - testing multiple functions together
class TestIntegration:
    """Test multiple functions working together."""
    
    def test_full_pipeline_simple(self):
        """Test a simple full pipeline from tokens to HTML."""
        # Start with token list
        tokens = ['<td>', '</td>', '<td>', '</td>']
        content = ['Header', 'Data']
        
        # Process through pipeline
        result = insert_text_to_token(tokens, content)
        html_result = htmlPostProcess(result)
        
        expected = '<html><body><table><td>Header</td><td>Data</td></tbody></table></body></html>'
        assert html_result == expected
    
    def test_full_pipeline_with_spans(self):
        """Test pipeline with span tokens."""
        tokens = ['<td', ' rowspan="2"', '>', '</td>', '<td>', '</td>']
        content = ['Spanned Cell', 'Normal Cell']
        
        result = insert_text_to_token(tokens, content)
        html_result = htmlPostProcess(result)
        
        expected = '<html><body><table><td rowspan="2">Spanned Cell</td><td>Normal Cell</td></tbody></table></body></html>'
        assert html_result == expected
    
    def test_full_pipeline_with_eb_tokens(self):
        """Test pipeline with empty bbox tokens."""
        tokens = ['<td>', '</td>', '<eb></eb>', '<td>', '</td>']
        content = ['Content1', 'Content2']
        
        result = insert_text_to_token(tokens, content)
        html_result = htmlPostProcess(result)
        
        expected = '<html><body><table><td>Content1</td><td></td><td>Content2</td></tbody></table></body></html>'
        assert html_result == expected
    
    def test_text_to_list_to_html(self):
        """Test from comma-separated string to final HTML."""
        input_text = "<td>cell1</td>,<td>cell2</td>"
        
        # Convert to list
        token_list = text_to_list(input_text)
        list_as_string = ','.join(token_list)
        html_result = htmlPostProcess(list_as_string)
        
        expected = '<html><body><table><td>cell1</td>,<td>cell2</td></table></body></html>'
        assert html_result == expected
    
    def test_deal_bb_with_duplicates_integration(self):
        """Test deal_bb with deal_duplicate_bb integration."""
        input_html = '<thead><tr><td><b>text<b>more</b>text</b></td></tr></thead>'
        
        result = deal_bb(input_html, 'thead')
        
        # Should handle both adding bold tags and fixing duplicates
        expected = '<thead><tr><td><b>textmoretext</b></td></tr></thead>'
        assert result == expected